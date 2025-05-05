<?php
/**
 * Widget Area Options Settings Module
 * Settings > Widget Options :: Widget Area Options
 *
 * @copyright   Copyright (c) 2017, Jeffrey Carandang
 * @since       3.5
*/

// Exit if accessed directly
if ( ! defined( 'ABSPATH' ) ) exit;

/**
 * Create Card Module for Widget Area Options
 *
 * @since 3.5
 * @global $widget_options
 * @return void
 */
if( !function_exists( 'widgetopts_settings_widget_area' ) ):
	function widgetopts_settings_widget_area(){
	    global $widget_options;
	    //avoid issue after update
	    if( !isset( $widget_options['widget_area'] ) ){
	        $widget_options['widget_area'] = '';
	    }

		$widget_area = ( isset( $widget_options['settings']['widget_area'] ) ) ? $widget_options['settings']['widget_area'] : array();?>
	    <li class="widgetopts-module-card <?php echo ( $widget_options['widget_area'] == 'activate' ) ? 'widgetopts-module-type-enabled' : 'widgetopts-module-type-disabled'; ?>" id="widgetopts-module-card-widget_area" data-module-id="widget_area">
			<div class="widgetopts-module-card-content">
				<h2><?php _e( 'Widget Area Options', 'widget-options' );?></h2>
				<p class="widgetopts-module-desc">
					<?php _e( 'Extra helpful management options below each sidebar widget areas.', 'widget-options' );?>
				</p>

				<div class="widgetopts-module-actions hide-if-no-js">
					<?php if( $widget_options['widget_area'] == 'activate' ){ ?>
						<button class="button button-secondary widgetopts-toggle-settings"><?php _e( 'Configure Settings', 'widget-options' );?></button>
						<button class="button button-secondary widgetopts-toggle-activation"><?php _e( 'Disable', 'widget-options' );?></button>
					<?php }else{ ?>
						<button class="button button-secondary widgetopts-toggle-settings"><?php _e( 'Learn More', 'widget-options' );?></button>
						<button class="button button-primary widgetopts-toggle-activation"><?php _e( 'Enable', 'widget-options' );?></button>
					<?php } ?>

				</div>
			</div>

			<?php widgetopts_modal_start( $widget_options['widget_area'] ); ?>
				<span class="dashicons widgetopts-dashicons widgetopts-no-top dashicons-art"></span>
				<h3 class="widgetopts-modal-header"><?php _e( 'Widget Area Options', 'widget-options' );?></h3>
				<p>
					<?php _e( 'Enable <strong>Remove All Widgets</strong> and/or <strong>Download Backup</strong> link options below each sidebar widget areas. This will help you manage your widgets better as always.', 'widget-options' );?>
				</p>

				<table class="form-table widgetopts-settings-section">
					<tr>
						<th scope="row">
							<label for="widgetopts-widget-area-remove"><?php _e( 'Remove Widgets Link', 'widget-options' );?></label>
						</th>
						<td>
							<input type="checkbox" id="widgetopts-widget-area-remove" name="widget_area[remove]" <?php echo widgetopts_is_checked( $widget_area, 'remove' ) ?> value="1" />
							<label for="widgetopts-widget-area-remove"><?php _e( 'Remove Widgets Link', 'widget-options' );?></label>
							<p class="description">
								<?php _e( 'Show "Remove All Widgets" link below each widget areas.', 'widget-options' );?>
							</p>
						</td>
					</tr>
					<tr>
						<th scope="row">
							<label for="widgetopts-widget-area-backup"><?php _e( 'Download Backup', 'widget-options' );?></label>
						</th>
						<td>
							<input type="checkbox" id="widgetopts-widget-area-backup" name="widget_area[backup]" <?php echo widgetopts_is_checked( $widget_area, 'backup' ) ?> value="1" />
							<label for="widgetopts-widget-area-backup"><?php _e( 'Enable Download Backup', 'widget-options' );?></label>
							<p class="description">
								<?php _e( 'Show "Download Backup" link below each sidebar widget area.', 'widget-options' );?>
							</p>
						</td>
					</tr>
				</table>
			<?php widgetopts_modal_end( $widget_options['widget_area'] ); ?>

		</li>
	    <?php
	}
	add_action( 'widgetopts_module_cards', 'widgetopts_settings_widget_area', 64 );
endif;
?>
