<?php
/**
 * Live Search Settings Module
 * Settings > Widget Options :: Live Widget Search
 *
 * @copyright   Copyright (c) 2017, Jeffrey Carandang
 * @since       3.3
*/

// Exit if accessed directly
if ( ! defined( 'ABSPATH' ) ) exit;

/**
 * Create Card Module for Live Widget Search
 *
 * @since 3.3
 * @global $widget_options
 * @return void
 */

 /*
 * Note: Please add a class "no-settings" in the <li> card if the card has no additional configuration, if there are configuration please remove the class
 */

if( !function_exists( 'widgetopts_settings_search' ) ):
	function widgetopts_settings_search(){
	    global $widget_options;
		//prevent undefined index error on upgrade
		if( !isset( $widget_options['search'] ) ){
			$widget_options['search'] = '';
		}
		?>
		<li class="widgetopts-module-card widgetopts-module-card-no-settings no-settings <?php echo ( $widget_options['search'] == 'activate' ) ? 'widgetopts-module-type-enabled' : 'widgetopts-module-type-disabled'; ?>" id="widgetopts-module-card-search" data-module-id="search">
			<div class="widgetopts-module-card-content">
				<h2><?php _e( 'Live Widget Search', 'widget-options' );?></h2>
				<p class="widgetopts-module-desc">
					<?php _e( 'Add live widget and sidebar search option on widgets.php admin dashboard.', 'widget-options' );?>
				</p>

				<div class="widgetopts-module-actions hide-if-no-js">
					<?php if( $widget_options['search'] == 'activate' ){ ?>
						<button class="button button-secondary widgetopts-toggle-settings"><?php _e( 'Learn More', 'widget-options' );?></button>
						<button class="button button-secondary widgetopts-toggle-activation"><?php _e( 'Disable', 'widget-options' );?></button>
					<?php }else{ ?>
						<button class="button button-secondary widgetopts-toggle-settings"><?php _e( 'Learn More', 'widget-options' );?></button>
						<button class="button button-primary widgetopts-toggle-activation"><?php _e( 'Enable', 'widget-options' );?></button>
					<?php } ?>

				</div>
			</div>

			<?php widgetopts_modal_start( $widget_options['search'] ); ?>
				<span class="dashicons widgetopts-dashicons widgetopts-no-top dashicons-search"></span>
				<h3 class="widgetopts-modal-header"><?php _e( 'Live Widget & Sidebar Search', 'widget-options' );?></h3>
				<p>
					<?php _e( 'This feature will add search box before available widgets area that will let you filter the widgets for better widget handling. This will also add search box above the sidebar chooser when you click each widgets for you to assign them easily.', 'widget-options' );?>
				</p>
				<p class="widgetopts-settings-section">
					<?php _e( 'No additional settings available.', 'widget-options' );?>
				</p>
			<?php widgetopts_modal_end( $widget_options['search'] ); ?>

		</li>
	    <?php
	}
	add_action( 'widgetopts_module_cards', 'widgetopts_settings_search', 64 );
endif;
?>
