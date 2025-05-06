<?php
/**
 * Extra Functions
 *
 * Collections of extra functions to avoid repeatition
 *
 * @copyright   Copyright (c) 2016, Jeffrey Carandang
 * @since       4.0
 */

 //create separate function returning classes for reuse
if( !function_exists( 'widgetopts_classes_generator' ) ){
    function widgetopts_classes_generator( $opts, $tabs, $settings, $so = false ){
        if( !empty( $opts ) && is_array( $opts ) ){
            $classes        = array();
            $devices        = isset( $opts['devices'] )     ? $opts['devices'] : '';
            $alignment      = isset( $opts['alignment'] )   ? $opts['alignment'] : '';
            $columns        = isset( $opts['column'] )      ? $opts['column'] : '';
            $clearfix       = isset( $opts['clearfix'] )    ? $opts['clearfix'] : '';
            $custom_class   = isset( $opts['class'] )       ? $opts['class'] : '';
            $abbr           = array(
                                'mobile'    =>  'xs',
                                'tablet'    =>  'sm',
                                'desktop'   =>  'md',
                            );
            if( isset( $devices['options'] ) ){
                unset( $devices['options'] );
            }

            if( 'activate' == $tabs['devices'] ){
                //devices visibility
                if( !empty( $devices ) ){
                    $device_opts    = ( isset( $opts['devices']['options'] ) ) ? $opts['devices']['options'] : 'hide';
                    $classes[] = 'extendedwopts-' . $device_opts ;

                    foreach ($devices as $key => $value) {
                        $classes[] = 'extendedwopts-' . $key;
                    }
                }
            }

            if( 'activate' == $tabs['alignment'] ){
                //alignment
                if( !empty( $alignment ) ){
                    foreach ($alignment as $k => $v) {
                        if( 'default' != $v ){
                            $classes[] = 'extendedwopts-' . $abbr[ $k ] . '-'. $v ;
                        }
                    }
                }
            }

            if( 'activate' == $tabs['classes'] && isset( $settings['classes'] ) ){
                //classes & ID
                // $options    = get_option('extwopts_class_settings');
                $predefined = array();
                if( isset( $settings['classes'] ) && isset( $settings['classes']['classlists'] ) && !empty( $settings['classes']['classlists'] ) ){
                    $predefined = $settings['classes']['classlists'];
                }

                //don't add any classes when settings is set to predefined or hide
                if( !isset( $settings['classes']['type'] ) ||
                    ( isset(  $settings['classes']['type'] ) && !in_array(  $settings['classes']['type'] , array( 'hide', 'predefined' ) ) ) ){
                    if( is_array( $custom_class ) && isset( $custom_class['classes'] ) && !empty( $custom_class['classes'] ) ){
                        $classes[] = $custom_class['classes'];
                    }
                }

                //don't add any classes when settings is set to text or hide
                if( !isset(  $settings['classes']['type'] ) ||
                    ( isset(  $settings['classes']['type'] ) && !in_array( $settings['classes']['type'] , array( 'hide', 'text' ) ) ) ){
                    if( is_array( $predefined ) && !empty( $predefined ) ){
                        $predefined = array_unique( $predefined );
                        if( isset( $custom_class['predefined'] ) && is_array( $custom_class['predefined'] ) ){
                            $filtered = array_intersect( $predefined, $custom_class['predefined'] );
                            if( !empty( $filtered ) ){
                                $classes = array_merge( $classes,  $filtered );
                                // $classes[] = implode( ' ', $filtered );
                                // $classes[] = ' ';
                            }
                        }
                    }
                }
            }

            if( $so && 'activate' == $tabs['hide_title'] ){
                //add fixed class to widget
                if( isset( $custom_class['title'] ) && !empty( $custom_class['title'] ) ){
                    $classes[] = 'widgetopts-hide_title';
                }
            }

            return apply_filters( 'widgetopts_get_classes', $classes );
        }
    }
}

//add is_active_sidebar support
if( !function_exists( 'widgetopts_sidebars_widgets' ) ){
	add_action( 'wp_loaded', 'widgetopts_sidebars_widgets_action' );
	function widgetopts_sidebars_widgets_action() {
        if( apply_filters( 'widgetopts_is_active_sidebar_support', true ) ){
    		add_filter( 'sidebars_widgets', 'widgetopts_sidebars_widgets' );
        }
	}
	function widgetopts_sidebars_widgets( $sidebars ) {
		if ( is_admin() ) {
			return $sidebars;
		}

		global $wp_registered_widgets;
        $checked = array();

		foreach ( $sidebars as $s => $sidebar ) {
			if ( $s == 'wp_inactive_widgets' || strpos( $s, 'orphaned_widgets' ) === 0 || empty( $sidebar ) ) {
				continue;
			}

			foreach ( $sidebar as $w => $widget ) {
				// $widget is the id of the widget
				if ( ! isset( $wp_registered_widgets[ $widget ] ) ) {
					continue;
				}

				if ( isset( $checked[ $widget ] ) ) {
					$show = $checked[ $widget ];
				} else {
					$opts = $wp_registered_widgets[ $widget ];
					$id_base = is_array( $opts['callback'] ) || $opts['callback'] instanceof ArrayAccess ? $opts['callback'][0]->id_base : $opts['callback'];

					if ( ! is_string( $id_base ) ) {
						continue;
					}

					$instance = get_option( 'widget_' . $id_base );

					if ( ! $instance || ! is_array( $instance ) ) {
						continue;
					}

					if ( isset( $instance['_multiwidget'] ) && $instance['_multiwidget'] ) {
						$number = $opts['params'][0]['number'];
						if ( ! isset( $instance[ $number ] ) ) {
							continue;
						}

						$instance = $instance[ $number ];
						unset( $number );
					}

					unset( $opts );

					$show = widgetopts_display_callback( $instance, (object) array( 'id' => $widget ), '' );

					$checked[ $widget ] = $show ? true : false;
				}

				if ( ! $show ) {
					unset( $sidebars[ $s ][ $w ] );
				}

				unset( $widget );
			}
			unset( $sidebar );
		}

		return $sidebars;
	}
}

?>
